import pandas as pd
import pickle as pk
import streamlit as st
import time
import base64

# Load model and data
model = pk.load(open(
    r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Gold_Price_Prediction\Gold_price.pkl', 'rb'
))
data = pd.read_csv(
    r'C:\Users\quant\Projects\ML_Projects\Datasets\Supervised\Regression\gld_price_data.csv'
)

# Convert background image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

# Fix: Only pass image path, not mode
bg_base64 = get_base64_image(r"C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Gold_Price_Prediction\Gold.jpeg")

# Set Streamlit page config
st.set_page_config(page_title='Gold Price Predictor 🪙', layout='centered')

# Custom background and styles
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{bg_base64}");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    .title {{
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #000;
    }}

    .predict-box {{
        background: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #00e676;
        color: #00e676;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        animation: fadeIn 1s ease-in;
    }}

    .predict-box h2 {{
        color: #4CAF50;
        font-size: 28px;
    }}
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<div class='title'>🪙 Gold Price Predictor</div>", unsafe_allow_html=True)

# Input sliders
SPX = st.slider('📈 S&P 500 Index', min_value=int(data['SPX'].min()), max_value=int(data['SPX'].max()), value=int(data['SPX'].mean()))
USO = st.slider('🛢️ US Oil Price (USO)', float(data['USO'].min()), float(data['USO'].max()), float(data['USO'].mean()))
SLV = st.slider('🥈 Silver Price (SLV)', float(data['SLV'].min()), float(data['SLV'].max()), float(data['SLV'].mean()))
EUR_USD = st.slider('💱 EUR/USD Currency Pair', float(data['EUR/USD'].min()), float(data['EUR/USD'].max()), float(data['EUR/USD'].mean()))

# Collect input into DataFrame
inputs = pd.DataFrame([[SPX, USO, SLV, EUR_USD]], columns=['SPX', 'USO', 'SLV', 'EUR_USD'])

# Predict button
if st.button("🔮 Predict Gold Price"):
    st.toast("Predicting Gold Price...", icon="⌛")
    time.sleep(1)
    try:
        prediction = model.predict(inputs)[0]
        st.markdown(f"""
            <div class='predict-box'>
                <h2>💸 Estimated Gold Price</h2>
                <p>{round(prediction, 2):,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❗ Error during prediction: {e}")

# About section
with st.expander("ℹ️ About this App"):
    st.markdown("""
        This application uses a machine learning model to predict gold prices based on:
        - 📈 S&P 500 index (SPX)
        - 🛢️ Oil price (USO)
        - 🥈 Silver price (SLV)
        - 💱 Currency exchange rate (EUR/USD)

        It is trained using historical data to assist in:
        - 📊 Economic forecasting
        - 💰 Investment strategy
        - 📈 Financial analysis

        👨‍💻 Developed by **QubitSpace**
    """)
