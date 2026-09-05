import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="AI Flood Risk Dashboard", layout="wide")
model = joblib.load("flood_model.pkl")
model_columns = joblib.load("model_columns.pkl")

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🌧️ AI-Based Flood Risk Prediction Dashboard")
    st.caption("Predicting flood probability using Random Forest Algorithm")
with col2:
    st.write(datetime.now().strftime("Date: %d %b %Y | Time: %H:%M"))

st.sidebar.header("📥 INPUT PARAMETERS")
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 600.0, 127.0)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 48.0, 28.0)
humidity = st.sidebar.slider("Humidity (%)", 15.0, 100.0, 60.0)
river_discharge = st.sidebar.slider("River Discharge (m³/s)", 0.0, 6000.0, 1564.0)
water_level = st.sidebar.slider("Water Level (m)", 0.0, 12.0, 3.6)
elevation = st.sidebar.slider("Elevation (m)", 1.0, 8391.0, 544.0)
land_cover = st.sidebar.selectbox("Land Cover", ['Agricultural', 'Urban', 'Forest', 'Desert', 'Water Body'])
soil_type = st.sidebar.selectbox("Soil Type", ['Loam', 'Clay', 'Sandy', 'Silt', 'Peat'])
population_density = st.sidebar.slider("Population Density", 2.0, 10000.0, 2078.0)
infrastructure = st.sidebar.selectbox("Infrastructure (0=Poor,1=Good)", [0, 1])
historical_floods = st.sidebar.selectbox("Historical Floods (past)", [0, 1])
area = st.sidebar.number_input("Area (m2)", value=35000000)

predict_btn = st.sidebar.button("🔍 Predict Flood Risk")

if predict_btn:
    input_dict = {
        'Rainfall (mm)': rainfall,
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'River Discharge (m³/s)': river_discharge,
        'Water Level (m)': water_level,
        'Elevation (m)': elevation,
        'Area (m2)': area,
        'Population Density': population_density,
        'Infrastructure': infrastructure,
        'Historical Floods': historical_floods,
    }
    for lc in ['Agricultural', 'Desert', 'Forest', 'Urban', 'Water Body']:
        input_dict[f'Land Cover_{lc}'] = 1 if land_cover == lc else 0
    for s in ['Clay', 'Loam', 'Peat', 'Sandy', 'Silt']:
        input_dict[f'Soil Type_{s}'] = 1 if soil_type == s else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    prob = model.predict_proba(input_df)[0][1] * 100
    st.metric("Flood Probability", f"{prob:.1f}%")
