import os
import warnings
import logging
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from tensorflow.keras.models import load_model
import pandas as pd
import joblib

# Suppress TensorFlow and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Load data and models
@st.cache_data
def load_data_and_model():
    file_path_2022 = 'ChallengePOLITO_MastriniMagazzino_2022_1.xls'
    df_2022 = pd.read_excel(file_path_2022, engine='xlrd')

    file_path_2023 = 'ChallengePOLITO_MastriniMagazzino_2023_1.xls'
    df_2023 = pd.read_excel(file_path_2023, engine='xlrd')

    file_path_2024 = 'ChallengePOLITO_MastriniMagazzino_2024_1.xls'
    df_2024 = pd.read_excel(file_path_2024, engine='xlrd')

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
    df = df[['product_name', 'initialstock', 'finalstock', 'movementdate', 'qty', 'uom']]
    label_encoder = LabelEncoder()
    df['product_encoded'] = label_encoder.fit_transform(df['product_name'])

    model = load_model('model.h5')
    scaler_y = joblib.load('scaler_y.pkl')
    scaler_X = joblib.load('scaler_X.pkl')

    return df, label_encoder, model, scaler_y, scaler_X

df, label_encoder, model, scaler_y, scaler_X = load_data_and_model()

# Function to predict quantity
def predict_quantity(product_name, month, df, label_encoder, model, scaler_y, scaler_X):
    product_encoded = label_encoder.transform([product_name])[0]
    season = (month % 12 + 3) // 3
    week_of_year = (month - 1) * 4 + 2
    initialstock_mean = df[df['product_name'] == product_name]['initialstock'].mean()
    finalstock_mean = df[df['product_name'] == product_name]['finalstock'].mean()

    X_new = np.array([[product_encoded, month, 0, season, week_of_year, initialstock_mean, finalstock_mean]])
    X_new_scaled = scaler_X.transform(X_new)  # Use pre-fitted scaler_X

    y_pred_norm = model.predict(X_new_scaled).flatten()
    y_pred_denorm_log = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Invert logarithmic transformation
    predicted_qty = np.expm1(y_pred_denorm_log)
    
    # Check the unit of measure
    uom = df[df['product_name'] == product_name]['uom'].iloc[0]
    if uom == "Numero" or pd.isna(uom):  # Treat missing "uom" as "Numero"
        return np.ceil(predicted_qty).astype(int)  # Round up to the nearest integer
    elif uom == "MT" or uom == "KG":
        return predicted_qty  # Return as float
    else:
        raise ValueError(f"Unknown unit of measure: {uom}")

# Months mapping
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# CSS for layout and background
st.markdown("""
    <style>
        body {
            background-color: #FFF9E3; /* Light yellow background */
        }
        .stApp {
            background: #FFF9E3; /* Light yellow background */
        }
        .output-box {
            background-color: #FFE599;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            color: #784212;
            font-weight: bold;
        }
        input {
            font-size: 18px;
            padding: 8px;
        }    
        .logo-container {
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Add the company logo
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("logo_zeca_ita.jpg", width=150)
st.markdown('</div>', unsafe_allow_html=True)

# App title
st.title("📦 Warehouse Forecast")
st.write("Enter the product name and the month to get the required quantity for the warehouse.")

# Dynamic input for product name with suggestions, sorted alphabetically
search_query = st.text_input("🔍 Type the product name:")
matching_products = np.sort(df[df['product_name'].str.contains(search_query, case=False, na=False)]['product_name'].unique())

if len(matching_products) > 0:
    product_name = st.selectbox("Select a product:", matching_products)
else:
    product_name = None
    st.info("No product found, try typing another name.")

# Input for month
month_name = st.selectbox("📅 Select the month:", list(MONTHS.keys()))
month = MONTHS[month_name]

# Button to predict
if st.button("📊 Predict") and product_name:
    result = predict_quantity(product_name, month, df, label_encoder, model, scaler_y, scaler_X)
    st.markdown(f"""
        <div class="output-box">
            Predicted quantity for the product <b>{product_name}</b> in the month of <b>{month_name}</b>: <br><br>
            <span style="font-size: 30px;">{result[0]:.2f}</span>
        </div>
    """, unsafe_allow_html=True)
