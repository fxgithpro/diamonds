import streamlit as st
import requests
import os
from rate_change import load_exchange_rates, convert_price, get_currency_symbol

# Load exchange rates
try:
    exchange_rates = load_exchange_rates()
except Exception as e:
    #st.error(f"Failed to load exchange rates: {e}")
    exchange_rates = {'USD': 1.0}  # Fallback

# API base URL from environment variable (defaults to localhost if not set)
#API_BASE_URL = os.environ.get("API_BASE_URL", "https://diamonds-861302064365.europe-west1.run.app")
API_BASE_URL = "https://diamonds-861302064365.europe-west1.run.app"
API_URL = f"{API_BASE_URL}/predict_one"

# Categorical options based on standard diamonds dataset
CUT_OPTIONS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_OPTIONS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_OPTIONS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

st.title("Diamond Price Predictor")
st.write("Enter diamond characteristics to get a price prediction from the REST API.")

# Input form
with st.form("diamond_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        carat = st.number_input("Carat", min_value=0.0, step=0.01, format="%.2f")
        cut = st.selectbox("Cut", CUT_OPTIONS)
        color = st.selectbox("Color", COLOR_OPTIONS)
        clarity = st.selectbox("Clarity", CLARITY_OPTIONS)
        currency = st.selectbox("Display Currency", list(exchange_rates.keys()))
    
    with col2:
        depth = st.number_input("Depth (%)", min_value=0.0, step=0.1, format="%.1f")
        table = st.number_input("Table (%)", min_value=0.0, step=0.1, format="%.1f")
        x = st.number_input("Length (x) in mm", min_value=0.0, step=0.01, format="%.2f")
        y = st.number_input("Width (y) in mm", min_value=0.0, step=0.01, format="%.2f")
        z = st.number_input("Depth (z) in mm", min_value=0.0, step=0.01, format="%.2f")
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Prepare payload matching Diamant model
    payload = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            converted_price = convert_price(result['price'], currency, exchange_rates)
            symbol = get_currency_symbol(currency)
            st.success(f"Predicted Price: {symbol}{converted_price:.2f}")
            st.json(result)  # Optional: Show full response
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
