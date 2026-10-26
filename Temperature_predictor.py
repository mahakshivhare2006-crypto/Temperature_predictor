import streamlit as st
import pandas as pd
import joblib.load("Temperature_predictor_model.pkl")
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Temperature Predictor",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD ML MODEL ----------------
MODEL_PATH = "temperature_prediction_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------------- CUSTOM CSS ----------------
def apply_theme(theme):
    themes = {
        "winter": {
            "background": "linear-gradient(135deg, #0f2027, #203a63, #4ca1af)",
            "card": "rgba(255, 255, 255, 0.12)",
            "accent": "#9be7ff",
            "text": "#ffffff"
        },
        "pleasant": {
            "background": "linear-gradient(135deg, #134e5e, #71b280)",
            "card": "rgba(255, 255, 255, 0.15)",
            "accent": "#d5ffdc",
            "text": "#ffffff"
        },
        "summer": {
            "background": "linear-gradient(135deg, #ff8008, #ffc837)",
            "card": "rgba(255, 255, 255, 0.25)",
            "accent": "#fff7d6",
            "text": "#ffffff"
        }
    }

    selected = themes[theme]

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {selected['background']};
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, p, label, span {{
            color: {selected['text']};
        }}

        .hero {{
            text-align: center;
            padding: 25px 10px;
        }}

        .hero-title {{
            font-size: 42px;
            font-weight: 800;
            color: white;
        }}

        .hero-subtitle {{
            font-size: 17px;
            color: white;
            opacity: 0.9;
        }}

        .weather-card {{
            background: {selected['card']};
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 22px;
            padding: 28px;
            text-align: center;
            margin: 15px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }}

        .weather-icon {{
            font-size: 75px;
        }}

        .temperature {{
            font-size: 58px;
            font-weight: 800;
            color: white;
        }}

        .weather-status {{
            font-size: 25px;
            font-weight: 700;
            color: {selected['accent']};
        }}

        .weather-description {{
            font-size: 16px;
            color: white;
        }}

        div.stButton > button {{
            width: 100%;
            border-radius: 12px;
            min-height: 48px;
            font-size: 17px;
            font-weight: 700;
            background: white;
            color: #17324d;
            border: none;
            transition: 0.25s;
        }}

        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 18px rgba(0,0,0,0.2);
        }}

        div[data-testid="stForm"] {{
            background: rgba(255,255,255,0.10);
            padding: 22px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.22);
        }}

        .footer {{
            text-align: center;
            padding: 25px 0 5px;
            color: white;
            font-size: 13px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌍 Temperature Predictor</div>
        <div class="hero-subtitle">
            Smart Weather Estimation using Machine Learning
        </div>
        <p>Powered by Multiple Linear Regression</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT SECTION ----------------
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("## 🌡️ Weather Details")
    st.write("Enter the weather parameters to estimate temperature.")

    with st.form("weather_prediction_form"):

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        precipitation = st.number_input(
            "🌧️ Precipitation (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        wind_speed = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "✨ Predict Temperature"
        )


# ---------------- PREDICTION SECTION ----------------
with right:

    st.markdown("## 🌤️ Prediction Result")

    if submitted:
        try:
            if not os.path.exists(MODEL_PATH):
                st.error(
                    f"Model file not found: {MODEL_PATH}. "
                    "Please check your project files."
                )
                st.stop()

            model = load_model()

            # Keep the same feature names and order used in training.
            input_data = pd.DataFrame(
                [[humidity, precipitation, wind_speed]],
                columns=[
                    "Humidity_pct",
                    "Precipitation_mm",
                    "Wind_Speed_kmh"
                ]
            )

            prediction = float(model.predict(input_data)[0])

            if not pd.notna(prediction):
                st.error("The model returned an invalid prediction.")
                st.stop()

            # Display theme based on predicted temperature.
            # These are UI categories, not official weather warnings.
            if prediction <= 10:
                theme = "winter"
                icon = "❄️"
                status = "Winter Vibes"
                description = (
                    "The predicted temperature is low. "
                    "The interface has switched to a cool winter theme."
                )

            elif prediction < 25:
                theme = "pleasant"
                icon = "🌤️"
                status = "Pleasant Weather"
                description = (
                    "The predicted temperature is moderate. "
                    "Enjoy the refreshing weather theme!"
                )

            else:
                theme = "summer"
                icon = "☀️"
                status = "Summer Vibes"
                description = (
                    "The predicted temperature is high. "
                    "The interface has switched to a warm summer theme."
                )

            # Apply the matching theme to the page.
            apply_theme(theme)

            st.markdown(
                f"""
                <div class="weather-card">
                    <div class="weather-icon">{icon}</div>
                    <div class="temperature">
                        {prediction:.2f} °C
                    </div>
                    <div class="weather-status">{status}</div>
                    <br>
                    <div class="weather-description">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Input summary
            st.markdown("### 📊 Your Input Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Humidity", f"{humidity:.1f}%")
            col2.metric("Precipitation", f"{precipitation:.1f} mm")
            col3.metric("Wind Speed", f"{wind_speed:.1f} km/h")

            st.info(
                "Note: This is a machine-learning estimate based on "
                "your trained model. It is not a live weather forecast."
            )

        except Exception as e:
            st.error(
                "Prediction could not be completed. "
                "Check your model file, feature names, and input format."
            )
            st.caption(f"Technical details: {e}")

    else:
        apply_theme("pleasant")

        st.markdown(
            """
            <div class="weather-card">
                <div class="weather-icon">🌦️</div>
                <h3>Ready to Predict?</h3>
                <p class="weather-description">
                    Enter your weather details and click the button
                    to discover the predicted temperature.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### ✨ What you will see")
        st.write("❄️ A cool winter-inspired interface")
        st.write("🌤️ A refreshing pleasant-weather interface")
        st.write("☀️ A warm summer-inspired interface")


# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Temperature Prediction Project |
        Machine Learning · Multiple Linear Regression
    </div>
    """,
    unsafe_allow_html=True
)
