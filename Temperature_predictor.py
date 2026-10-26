import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("temperature_prediction_model.pkl")

# App title
st.title("🌡️ Temperature Prediction")
st.write("Predict temperature using weather conditions.")

st.header("Enter Weather Details")

# Humidity slider
humidity = st.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

# Precipitation slider
precipitation = st.slider(
    "Precipitation (mm)",
    min_value=0.0,
    max_value=500.0,
    value=10.0,
    step=0.5
)

# Wind Speed slider
wind_speed = st.slider(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=150.0,
    value=10.0,
    step=0.5
)

# Predict button
if st.button("Predict Temperature"):
    input_data = pd.DataFrame({
        "Humidity_pct": [humidity],
        "Precipitation_mm": [precipitation],
        "Wind_Speed_kmh": [wind_speed]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Temperature: {prediction[0]:.2f} °C"
    )
