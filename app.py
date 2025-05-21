import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

encoders = joblib.load("label_encoder.pkl")

# Title
st.title("🚗 Used Car Price Predictor. Its a CI/CD!")

# Input form
with st.form("car_form"):
    car_name = st.selectbox("Car Name", encoders["Car_Name"].classes_)
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, value=5.5)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100, value=40000)
    fuel_type = st.selectbox("Fuel Type", encoders["Fuel_Type"].classes_)
    seller_type = st.selectbox("Seller Type", encoders["Seller_Type"].classes_)
    transmission = st.selectbox("Transmission", encoders["Transmission"].classes_)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
    
    submitted = st.form_submit_button("Predict Selling Price")

# Prediction
if submitted:
    sample = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner
    }])
    
    # Apply encoding
    for col in sample.columns:
        if col in encoders and sample[col].dtype == 'object':
            sample[col] = encoders[col].transform(sample[col])

    # Ensure feature order matches training
    sample = sample[model.feature_names_in_]

    # Predict
    price = model.predict(sample)[0]
    st.success(f"💰 Estimated Selling Price: ₹{price:,.2f} Lakhs")
