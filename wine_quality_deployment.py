import streamlit as st
import pandas as pd
import joblib


model = joblib.load("wine_quality_prediction.pkl")

st.title("🍷 Wine Quality Prediction System")


col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=0.0)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.0)
    citric_acid = st.number_input("Citric Acid", value=0.0)
    residual_sugar = st.number_input("Residual Sugar", value=0.0)
    chlorides = st.number_input("Chlorides", value=0.0)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=0.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=0.0)
    density = st.number_input("Density", value=0.0)
    pH = st.number_input("pH", value=0.0)
    sulphates = st.number_input("Sulphates", value=0.0)
    alcohol = st.number_input("Alcohol", value=0.0)
    wine_type = st.selectbox("Wine Type (0=White, 1=Red)", options=[0, 1])


df = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
    "type": [wine_type],
    "quality":[0]
})

if st.button("Predict Quality"):
    try:
      
        df = df[model.feature_names_in_]
        
        prediction = model.predict(df)
        st.success(f"### Predicted Wine Quality: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Expected features by model:", model.feature_names_in_)
