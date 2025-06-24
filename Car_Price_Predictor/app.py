import pandas as pd
import streamlit as st
import pickle as pk
import numpy as np
import base64
import time

# Load model
model = pk.load(open(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Car_Price_Prediction\car_price_prediction.pkl', 'rb'))

# Load data
data = pd.read_csv(r'C:\Users\quant\Projects\ML_Projects\Datasets\Supervised\Regression\Cardetails.csv')
data['name'] = data['name'].apply(lambda x: x.split(' ')[0].strip())

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert your local background image to base64
bg_base64 = get_base64_image(r"C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Car_Price_Prediction\vCqmep.jpg")

# Page settings
st.set_page_config(page_title="Car Price Predictor 🏎️", layout="centered")

# CSS with base64 background image
st.markdown(f"""
    <style>
    html, body, .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    .card {{
        background: rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem auto;
        max-width: 750px;
        color: white;
        animation: fadeIn 1s ease-in;
    }}

    h1 {{
        color: white;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }}

    .prediction {{
        font-size: 22px;
        font-weight: bold;
        color: #00e676;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #00e676;
        animation: fadeIn 1s ease-in;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .stButton>button {{
        background-color: #00acc1;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: #00838f;
        cursor: pointer;
    }}

    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1>Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <h6 style="text-align:center; font-size:16px; color:#ccc;">
            🤖 This car price predictor is powered by a machine learning model developed by Rocket based Company Known as QubitSpace.
    </h6>
</div>
""", unsafe_allow_html=True)

# Inputs
name = st.selectbox('🚘 Car Brand', sorted(data['name'].unique()))
year = st.slider('📅 Year of Manufacture', 1994, 2024)
km_driven = st.slider('🛣️ Kilometers Driven', 0, 500000)
fuel = st.selectbox('⛽ Fuel Type', data['fuel'].unique())
seller_type = st.selectbox('👤 Seller Type', data['seller_type'].unique())
transmission = st.selectbox('🔀 Transmission Type', data['transmission'].unique())
owner = st.selectbox('🔁 Previous Owners', data['owner'].unique())
mileage = st.slider('📏 Mileage (km/l)', 10, 40)
engine = st.slider('⚙️ Engine (CC)', 700, 5000)
max_power = st.slider('🚀 Max Power (BHP)', 0, 200)
seats = st.slider('🪑 Number of Seats', 2, 10)

# Predict button
if st.button("💰 Predict Car Price"):
    # Show toast immediately
    st.toast("Analysing Car Details...", icon="⌛")

    # Show spinner during processing
    with st.spinner("Analyzing Car Details..."):
        time.sleep(2)
        try:
            inputs = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
                columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])

            inputs['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                     'Fourth & Above Owner', 'Test Drive Car'],
                                    [1, 2, 3, 4, 5], inplace=True)
            inputs['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
            inputs['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
            inputs['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
            inputs['name'].replace(
                ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                list(range(1, 32)), inplace=True)

            price = model.predict(inputs)[0]
            st.markdown(
                f'<div class="prediction">💸 Estimated Car Price<br> ₹{round(price, 2):,.2f}</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# End card
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("ℹ️ About this App"):
    st.markdown("""
    - 🚗 Developed by QubitSpace using machine learning and real-world car pricing data.
    - 📉 Predicts car resale values based on brand, age, mileage, fuel type, and more.
    - 🧠 Designed for car buyers, sellers, and auto industry analysts seeking price insights.
    """)


