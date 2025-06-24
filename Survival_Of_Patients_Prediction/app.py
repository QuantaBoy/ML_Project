import streamlit as st
import pandas as pd
import pickle
import base64
import time

st.set_page_config(page_title="Patient Survival Predictor🩺", layout="centered")
# --------- Function to Encode Local Image as Base64 ---------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# --------- Local Image Path ---------
local_image_path = r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Classification\Survival_Of_Patients_Prediction\noveyank_a_minimalistic_background_using_black_and_dark_flat_go_1edb64db-cab1-4ea4-99b4-fae139a6bd1d.jpg'
base64_image = get_base64_image(local_image_path)

# --------- Inject CSS for Background and Styling ---------
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{base64_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1 {{
            color: #ffffff;
            text-align: center;
            font-size: 2.8rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 2rem;
        }}
        .card {{
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(5px);
            border-radius: 20px;
            padding: 1rem;
            margin: 1rem auto;
            max-width: 750px;
            text-align:center;
            color: white;
        }}
        .glass {{
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
        }}
        .glass2 {{
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(5px);
            border-radius: 20px;
            padding: 1rem 2rem;
            margin: 1rem auto;
            max-width: 800px;
            color: #ccc;
            text-align: center;
            animation: fadeIn 1s ease-in;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}
        label {{
            color: #ffffff !important;
            font-weight: bold;
        }}
        .stButton>button {{
            background-color: #00acc1;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.5em 1.5em;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #008b9a;
            cursor: pointer;
        }}
        .st-success, .st-error {{
            font-size: 1.2rem;
            font-weight: bold;
        }}
        @keyframes fadeIn {{
        0% {{
            opacity: 0;
            transform: translateY(20px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
            }}
        }}
    </style>
""", unsafe_allow_html=True)

# --------- Load Model and Encoders ---------
with open(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Classification\Survival_Of_Patients_Prediction\model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'C:\Users\quant\Projects\ML_Projects\ML_model\Supervised\Classification\Survival_Of_Patients_Prediction\encoders.pkl', 'rb') as f:
    le_drugs, le_smoker, le_place = pickle.load(f)

# --------- Title ---------
st.markdown("""
    <div>
        <h1>Pharma Patient 1-Year Survival Predictor 🧬</h1>
    </div>
    <div class="glass">
        <h6 style='color: white; margin: 0;color:#ccc;'>
            🤖 This predictor estimates patient survival likelihood using a machine learning model developed by QubitSpace.
        </h6>
    </div>
""", unsafe_allow_html=True)

# --------- Input Section ---------
st.markdown("<div><h4>Enter Patient Details 📋</h4></div>", unsafe_allow_html=True)

Diagnosed_Condition = st.slider("Diagnosed Condition 🩺", 0, 52)
Patient_Age = st.number_input("Patient Age 🎂", min_value=0)
Patient_Body_Mass_Index = st.number_input("Body Mass Index ⚖️", min_value=0.0)
Number_of_prev_cond = st.slider("Number of Previous Conditions 📖", 0, 5)

col1, col2, col3 = st.columns(3)
with col1:
    drugs = st.selectbox("💊 Treated With Drugs", le_drugs.classes_)
with col2:
    smoker = st.selectbox("🚬 Patient Smoker", le_smoker.classes_)
with col3:
    place = st.selectbox("🌍 Patient Location", le_place.classes_)

# --------- Prediction Section ---------
if st.button("🔮 Predict Survival Rate"):
    # Show toast immediately
    st.toast("Analysing Patient Details...", icon="⌛")

    # Show spinner during processing
    with st.spinner("Analyzing Patient Details..."):
        time.sleep(2)
        try:
            input_df = pd.DataFrame([{
                'Diagnosed_Condition': Diagnosed_Condition,
                'Patient_Age': Patient_Age,
                'Patient_Body_Mass_Index': Patient_Body_Mass_Index,
                'Number_of_prev_cond': Number_of_prev_cond,
                'drugs_n': le_drugs.transform([drugs])[0],
                'smoker_n': le_smoker.transform([smoker])[0],
                'Place_n': le_place.transform([place])[0]
            }])

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.markdown("""
                    <div class = "glass2">
                        <h5 style="color:#ccc;">Model Prediction : The patient is likely to survive for 1 year ✅</h5>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class = "glass2">
                        <h2 style="color:lightred">The patient is unlikely to survive for 1 year ⚠️</h2>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --------- About Section ---------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - Built by QubitSpace 🧠 using clinical and pharmaceutical data  
    - Predicts patient 1-year survival likelihood 🧬 based on medical and lifestyle factors  
    - Designed for researchers, clinicians, and data scientists 💉  
    - Uses machine learning to help identify patent risk 🚀  
    """)
