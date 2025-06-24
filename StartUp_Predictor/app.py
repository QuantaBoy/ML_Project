import streamlit as st
import pandas as pd
import pickle as pk
import base64
import time

# Load the trained model
model = pk.load(open(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Classification\StartUp_Prediction\Startup_Predictor.pkl', 'rb'))

# Convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')

# Background image base64
bg_base64 = get_base64_image(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Classification\StartUp_Prediction\pngtree-person-is-holding-a-glass-ball-with-its-hands-picture-image_3428698.jpg')

# Page config
st.set_page_config(page_title="Startup Predictor 🚀", layout="centered")

# Custom background CSS with overlay card style
st.markdown(f"""
    <style>
    html, body, .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    .stButton>button {{
        background-color: #00acc1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: #008b9a;
        cursor: pointer;
    }}  

    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .card {{
        background: rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 1rem 2rem;
        margin: 1rem auto;
        max-width: 800px;
        color: white;
        text-align: center;
        animation: fadeIn 1s ease-in;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}

    .glass {{
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
        border-radius: 20px;
        padding: 1rem 2rem;
        margin: 1rem auto;
        max-width: 650px;
        color: white;
        text-align: center;
        animation: fadeIn 1s ease-in;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}
    .card3 {{
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 1rem 2rem;
        max-width: 800px;
        color: #ccc;
        animation: fadeIn 1s ease-in;
    }}
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("""
    <div>
        <h1 style='color: white; margin: 0;text-align:center'>Startup Success Predictor</h1>
    </div>
""", unsafe_allow_html=True)

# Description
st.markdown("""
    <div class="glass">
        <h6 style='color: white; margin: 0;color:#ccc;'>
            🤖 This startup predictor is powered by a machine learning model developed by Mohammed Riyan.
        </h6>
    </div>
""", unsafe_allow_html=True)

# Input Section
with st.container():
    st.subheader("Enter Company Details 📝")
    col1, col2 = st.columns(2)

    with col1:
        relationships = st.slider('👥 Relationships', 0, 63, 1)
        age_last_funding_year = st.slider('📅 Age Last Funding Year', -9.0466, 21.8959, 0.0)
        milestones = st.slider('🎯 Milestones Achieved', 0, 8, 1)
        is_software = st.selectbox('💻 Software Based Company', ['Yes', 'No'])
        is_web = st.selectbox('🌐 Web Based Company', ['Yes', 'No'])
        is_mobile = st.selectbox('📱 Mobile Based Company', ['Yes', 'No'])
        is_enterprise = st.selectbox('🏢 Enterprise Based Company', ['Yes', 'No'])

    with col2:
        is_advertising = st.selectbox('📣 Advertising Based Company', ['Yes', 'No'])
        is_gamesvideo = st.selectbox('🎮🎥 Game/Video Based Company', ['Yes', 'No'])
        is_ecommerce = st.selectbox('🛒 Ecommerce Based Company', ['Yes', 'No'])
        is_biotech = st.selectbox('🧬 Biotech Based Company', ['Yes', 'No'])
        is_consulting = st.selectbox('👨‍💻 Consultancy Based Company', ['Yes', 'No'])
        is_othercategory = st.selectbox('🧭 Other Based Company', ['Yes', 'No'])
        is_top500 = st.selectbox('🏆 Top 500 Company', ['Yes', 'No'])

# Map Yes/No to binary
yes_no_map = {'Yes': 1, 'No': 0}

# Prepare input for prediction
input_dict = {
    'age_last_funding_year': age_last_funding_year,
    'relationships': relationships,
    'milestones': milestones,
    'is_software': yes_no_map[is_software],
    'is_web': yes_no_map[is_web],
    'is_mobile': yes_no_map[is_mobile],
    'is_enterprise': yes_no_map[is_enterprise],
    'is_advertising': yes_no_map[is_advertising],
    'is_gamesvideo': yes_no_map[is_gamesvideo],
    'is_ecommerce': yes_no_map[is_ecommerce],
    'is_biotech': yes_no_map[is_biotech],
    'is_consulting': yes_no_map[is_consulting],
    'is_othercategory': yes_no_map[is_othercategory],
    'is_top500': yes_no_map[is_top500]
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("🔮 Predict Status"):
    # Show toast immediately
    st.toast("Analysing Company Details...", icon="⌛")

    # Show spinner during processing
    with st.spinner("Analyzing Company Details..."):
        time.sleep(2)
        try:
            prediction = model.predict(input_df)[0]
            result = "Successful 🌟" if prediction == 1 else "Failed ❌"
            color = "#2ecc71" if prediction == 1 else "#e74c3c"
            msg = "Likely to Succeed! 🚀" if prediction == 1 else "High Risk of Failure. 💥"

            st.markdown(f"""
                <div class="glass">
                    <h2 style='color:{color}; font-size: 2.2em;'>Model Prediction: {result}</h2>
                    <p style='font-size: 1.1em;'>{msg}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Extra Info
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - 🚀 Built by QubitSpace using machine learning and Crunchbase data.
    - 📊 Predicts startup outcomes based on early-stage signals.
    - 🧠 Useful for founders, investors, and data enthusiasts!
    """)
