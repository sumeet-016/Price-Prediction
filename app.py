import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps.get(col, {})).fillna(0)
        return X.values

@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_car_price_model.pkl")

    df = pd.read_csv('car_price_prediction_updated.csv')
    return model, df

# Page Config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

try:
    pipeline, data = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.info("Ensure 'random_forest_car_price_model.pkl' and the CSV are in the same folder.")
    st.stop()

st.title("🚗 Complete Car Price Prediction System")
st.markdown("Please fill in all details accurately for a precise estimation.")

with st.form("car_form"):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        manufacturers = sorted(data['Manufacturer'].unique())
        selected_mfr = st.selectbox("Manufacturer", manufacturers)
        

        mfr_mask = data['Manufacturer'] == selected_mfr
        filtered_models = sorted(data[mfr_mask]['Model'].unique())
        selected_model = st.selectbox("Model", filtered_models)

    with row1_col2:
        category = st.selectbox("Category", sorted(data['Category'].unique()))
        leather = st.selectbox("Leather Interior", ["Yes", "No"])
        fuel = st.selectbox("Fuel Type", sorted(data['Fuel type'].unique()))

    with row1_col3:
        gear = st.selectbox("Gear Box Type", sorted(data['Gear box type'].unique()))
        drive = st.selectbox("Drive Wheels", sorted(data['Drive wheels'].unique()))
        wheel = st.selectbox("Wheel (Steering)", sorted(data['Wheel'].unique()))

    st.divider()
    
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row2_col1:
        levy = st.number_input("Levy (Tax)", value=float(data['Levy'].median()))
        mileage = st.number_input("Mileage (km)", min_value=0, value=0)
        
    with row2_col2:
        cylinders = st.number_input("Cylinders", min_value=1.0, max_value=16.0, value=4.0)
        airbags = st.slider("Airbags Count", 0, 16, 4)

    with row2_col3:
        color = st.selectbox("Color", sorted(data['Color'].unique()))
        age = st.number_input("Car Age (Years)", min_value=0, max_value=100, value=5)

    predict_btn = st.form_submit_button("Predict Estimated Price", type="primary")

if predict_btn:
    input_df = pd.DataFrame([{
        'Levy': levy,
        'Manufacturer': selected_mfr,
        'Model': selected_model,
        'Category': category,
        'Leather interior': leather,
        'Fuel type': fuel,
        'Mileage': mileage,
        'Cylinders': cylinders,
        'Gear box type': gear,
        'Drive wheels': drive,
        'Wheel': wheel,
        'Color': color,
        'Airbags': airbags,
        'Age': age
    }])

    try:
        log_pred = pipeline.predict(input_df)
        final_price = np.expm1(log_pred)[0]
        
        st.balloons()
        st.success(f"### Estimated Market Price: **${final_price:,.2f}**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")