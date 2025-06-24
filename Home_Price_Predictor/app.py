import pandas as pd
import pickle as pk
import streamlit as st
import time
# Load model and data
model = pk.load(open(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Home_Price_Prediction\House_prediction_model.pkl', 'rb'))
data = pd.read_csv(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Regression\Home_Price_Prediction\Cleaned_data.csv')

# Set page config
st.set_page_config(page_title="Bangalore House Price Predictor 🏘️", layout="centered")

# Inject custom CSS for background and styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton>button {
        background-color: #00acc1;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #008b9a;
        cursor: pointer;
    }

    .glass {
        background: rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 1rem 2rem;
        margin: 1rem auto;
        max-width: 650px;
        color: white;
        text-align: center;
        animation: fadeIn 1s ease-in;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .card2 {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 1rem 2rem;
        margin: 2rem auto;
        width: 500px;
        height :140px;
        text-align: center;
        color: white;
        animation: fadeIn 1s ease-in;
    }
    .card3 {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 1rem 2rem;
        max-width: 800px;
        color: #ccc;
        animation: fadeIn 1s ease-in;
    }
        @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("""
    <div>
        <h1 style='color: white; margin: 0;text-align:center'>Bangalore House Price Predictor</h1>
    </div>
""", unsafe_allow_html=True)

# App description
st.markdown("""
    <div class="glass">
        <h6 style='color: white; margin: 0;color:#ccc;'>
            🤖 This predictor estimates property prices in Bangalore using a machine learning model developed by QubitSpace.
        </h6>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.title("🏡 Predict Home Price")
    loc = st.selectbox('📍 Choose Location', sorted(data['location'].unique()))
    sqft = st.number_input('📐 Total Sqft', min_value=300.0, step=10.0, format="%.1f")
    beds = st.number_input('🛏️ Bedrooms', min_value=1, step=1)
    bath = st.number_input('🛁 Bathrooms', min_value=1, step=1)
    balc = st.number_input('🚪 Balconies', min_value=0, step=1)

# Collect inputs into a DataFrame
input_df = pd.DataFrame([[loc, sqft, bath, balc, beds]],
                        columns=['location', 'total_sqft', 'bath', 'balcony', 'bedroom'])

# Prediction button
if st.sidebar.button("🔮 Predict Price"):
        st.toast("Calculating the Home Price...",icon='🧮')
        time.sleep(2)
        try:
            prediction = model.predict(input_df)[0]
            st.markdown(f"""
                <div class="card2">
                    <h2 style="color: lightgreen;">💰 Estimated Price</h2>
                    <p style="font-size: 1.5em;">₹ {prediction * 100000:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❗ Error during prediction: {e}")

# About section
with st.expander("ℹ️ About this App"):
    st.markdown("""
    <div class="card3">
        <ul style="text-align: left; font-size: 16px;">
            <li>🧠 Powered by a machine learning model trained on real Bangalore housing data.</li>
            <li>🏡 Predicts prices based on square footage, location, number of bedrooms, bathrooms, and balconies.</li>
            <li>🚀 Built by <strong>QubitSpace</strong>, a rocket-based data company.</li>
            <li>📊 Useful for home buyers, investors, and real estate analysts.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

