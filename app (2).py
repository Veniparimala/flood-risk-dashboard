import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Flood Risk Dashboard", layout="wide")

# ---------- Load model, columns, and dataset ----------
model = joblib.load("flood_model.pkl")
model_columns = joblib.load("model_columns.pkl")
data = pd.read_csv("flood_risk_dataset_india_realistic.csv")

# ---------- Header ----------
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🌧️ AI-Based Flood Risk Prediction Dashboard")
    st.caption("Predicting flood probability using Random Forest Algorithm")
with col2:
    st.write(datetime.now().strftime("Date: %d %b %Y | Time: %H:%M"))

# ---------- Sidebar inputs ----------
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
reset_btn = st.sidebar.button("🔄 Reset")


def build_input_row():
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
    row = pd.DataFrame([input_dict])
    return row[model_columns]


def risk_level(prob):
    if prob < 25:
        return "LOW RISK", "🟢", "#2ecc71"
    elif prob < 50:
        return "MODERATE RISK", "🟡", "#f1c40f"
    elif prob < 75:
        return "HIGH RISK", "🟠", "#e67e22"
    else:
        return "VERY HIGH RISK", "🔴", "#e74c3c"


def get_recommendations(rainfall, water_level, river_discharge, elevation, historical_floods):
    tips = []
    if rainfall > 200:
        tips.append("🌧️ Very high rainfall detected — stay alert for flash floods.")
    elif rainfall > 100:
        tips.append("🌦️ Moderate-heavy rainfall — monitor conditions closely.")
    if water_level > 6:
        tips.append("🌊 Water level critically high — avoid low-lying areas and basements.")
    elif water_level > 4:
        tips.append("💧 Water level rising — keep an eye on nearby rivers/drains.")
    if river_discharge > 3000:
        tips.append("🚨 High river discharge — alert local authorities.")
    if elevation < 300:
        tips.append("⛰️ Low elevation area — higher exposure to flooding, plan evacuation routes.")
    if historical_floods == 1:
        tips.append("📜 This area has a history of flooding — extra caution advised.")
    tips.append("📻 Stay updated with real-time weather forecasts.")
    if len(tips) <= 1:
        tips.insert(0, "✅ Conditions currently appear stable.")
    return tips


# ---------- Top metric row ----------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💧 Rainfall (mm)", f"{rainfall:.0f}")
m2.metric("🌡️ Temperature (°C)", f"{temperature:.0f}")
m3.metric("💦 Humidity (%)", f"{humidity:.0f}")
m4.metric("🌊 Water Level (m)", f"{water_level:.1f}")
m5.metric("🏙️ Population Density", f"{population_density:.0f}")

st.divider()

if predict_btn:
    input_row = build_input_row()
    prob = model.predict_proba(input_row)[0][1] * 100
    level_text, level_emoji, level_color = risk_level(prob)

    c1, c2, c3 = st.columns([1, 1, 1.3])

    # ---- Gauge chart ----
    with c1:
        st.subheader("Flood Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': level_color},
                'steps': [
                    {'range': [0, 25], 'color': '#d5f5e3'},
                    {'range': [25, 50], 'color': '#fcf3cf'},
                    {'range': [50, 75], 'color': '#fdebd0'},
                    {'range': [75, 100], 'color': '#fadbd8'},
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ---- Risk level card ----
    with c2:
        st.subheader("Flood Risk Level")
        st.markdown(f"### {level_emoji} :red[{level_text}]" if prob >= 50 else f"### {level_emoji} {level_text}")
        if prob >= 75:
            st.error("There is a very high chance of flood occurrence in the selected area.")
        elif prob >= 50:
            st.warning("There is a high chance of flood occurrence in the selected area.")
        elif prob >= 25:
            st.info("Moderate chance of flood occurrence — stay cautious.")
        else:
            st.success("Low chance of flood occurrence in the selected area.")

    # ---- Risk map ----
    with c3:
        st.subheader("Flood Risk Map (Regional Overview)")
        sample = data.sample(min(1500, len(data)), random_state=1)
        fig_map = px.scatter_geo(
            sample,
            lat='Latitude',
            lon='Longitude',
            color='Flood Occurred',
            color_continuous_scale=['green', 'red'],
            scope='asia',
            height=320,
        )
        fig_map.update_geos(lataxis_range=[6, 38], lonaxis_range=[66, 98])
        fig_map.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    c4, c5, c6 = st.columns([1, 1, 1])

    # ---- Feature importance chart ----
    with c4:
        st.subheader("Risk Factor Importance (Random Forest)")
        importances = pd.Series(model.feature_importances_, index=model_columns).sort_values()
        fig_imp = px.bar(importances, orientation='h')
        fig_imp.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---- Trend chart (vary rainfall) ----
    with c5:
        st.subheader("Flood Probability Trend (vs Rainfall)")
        rain_range = list(range(0, 601, 30))
        trend_probs = []
        for r in rain_range:
            row = build_input_row().copy()
            row['Rainfall (mm)'] = r
            trend_probs.append(model.predict_proba(row)[0][1] * 100)
        fig_trend = px.line(x=rain_range, y=trend_probs, markers=True,
                             labels={'x': 'Rainfall (mm)', 'y': 'Flood Probability (%)'})
        fig_trend.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Recommendations ----
    with c6:
        st.subheader("Recommendations")
        for tip in get_recommendations(rainfall, water_level, river_discharge, elevation, historical_floods):
            st.write(tip)

    st.info("ℹ️ This prediction is based on historical data and a machine learning model (Random Forest). "
            "Please use it as a decision support tool and validate with real-time observations.")
else:
    st.info("👈 Set your input parameters in the sidebar and click **Predict Flood Risk** to see results.")
