import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Car Price Predictor", layout="wide", page_icon="🚗")

# =====================================================
# CUSTOM TRANSFORMER (Must be defined for pickle/joblib)
# =====================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps.get(col, {})).fillna(0)
        return X.values

# =====================================================
# LOAD ASSETS
# =====================================================
@st.cache_resource
def load_assets():
    # Load the final tuned pipeline
    model = joblib.load("final_car_price_model.pkl") 
    
    # Load the dataset for dropdown options
    df = pd.read_csv("Model_Training.csv")

    # Clean strings and format Manufacturers
    categorical_cols = [
        'Manufacturer','Model','Category','Leather interior','Fuel type',
        'Gear box type','Drive wheels','Color', 'Turbo'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['Manufacturer'] = df['Manufacturer'].str.title()
    return model, df

try:
    pipeline, data = load_assets()
except Exception as e:
    st.error("Could not load model or data. Ensure 'final_car_price_model.pkl' and 'Model_Training.csv' are in the directory.")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.title("🚗 Car Price Predictor")
    st.markdown("Predict **used car market price** using a Tuned Ensemble Model.")
    st.divider()
    st.info("The model analyzes features like Age, Mileage, and Manufacturer frequency to estimate price.")
    st.markdown("---")
    st.markdown("👨‍💻 **Sumeet Kumar Pal**")
    st.markdown("[GitHub](https://github.com/sumeet-016)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/palsumeet/)")
# =====================================================
# MAIN UI - CASCADING FILTERS
# =====================================================
st.title("🚘 Used Car Price Prediction")
st.markdown("Fill in the details below to get an estimated market price.")

with st.form("prediction_form"):
    # --- CASCADING DROPDOWNS ---
    st.subheader("Selection 1: Identity")
    col1, col2, col3 = st.columns(3)

    with col1:
        manufacturers = sorted(data['Manufacturer'].unique())
        selected_mfr = st.selectbox("1. Manufacturer", manufacturers)
        mfr_filtered = data[data['Manufacturer'] == selected_mfr]

    with col2:
        available_categories = sorted(mfr_filtered['Category'].unique())
        selected_cat = st.selectbox("2. Category", available_categories)
        cat_filtered = mfr_filtered[mfr_filtered['Category'] == selected_cat]

    with col3:
        
        available_models = sorted(cat_filtered['Model'].unique())
        selected_model = st.selectbox("3. Model", available_models)

    st.divider()

    # --- SPECIFICATIONS ---
    st.subheader("Selection 2: Mechanicals")
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        fuel = st.selectbox("Fuel type", sorted(data['Fuel type'].unique()))
        gear = st.selectbox("Gear box type", sorted(data['Gear box type'].unique()))

    with col5:
        drive = st.selectbox("Drive wheels", sorted(data['Drive wheels'].unique()))
        color = st.selectbox("Color", sorted(data['Color'].unique()))

    with col6:
        engine = st.number_input("Engine Volume (L)", value=2.0, step=0.1)
        cylinders = st.number_input("Cylinders", value=float(data['Cylinders'].median()), step=1.0)

    with col7:
        leather = st.selectbox("Leather interior", ["No", "Yes"])
        turbo = st.selectbox("Turbo Engine", ["No", "Yes"])
    st.divider()

    # --- USAGE & CONDITION ---
    st.subheader("Selection 3: Usage History")
    col8, col9, col10, col11 = st.columns(4)

    with col8:
        levy = st.number_input("Levy", value=float(data['Levy'].median()))
    
    with col9:
        mileage = st.number_input("Mileage (km)", value=int(data['Mileage'].median()), min_value=0)

    with col10:
        airbags = st.slider("Airbags", 0, 16, 4)

    with col11:
        age = st.number_input("Vehicle Age (Years)", value=5, min_value=0, max_value=80)

    submit = st.form_submit_button("🔮 Calculate Market Value")

# =====================================================
# PREDICTION LOGIC
# =====================================================
if submit:
    mileage_intensity = round(mileage / (age + 1), 2)

    input_df = pd.DataFrame([{
        'Mileage': mileage,
        'Age': age,
        'Engine Volume': engine,
        'Levy': levy,
        'Cylinders': cylinders,
        'Airbags': airbags,
        'Mileage_Intensity': mileage_intensity,
        'Category': selected_cat,
        'Fuel type': fuel,
        'Gear box type': gear,
        'Drive wheels': drive,
        'Color': color,
        'Leather interior': leather,
        'Turbo': turbo,
        'Manufacturer': selected_mfr,
        'Model': selected_model
    }])

    #  Predict using the Pipeline
    try:
        log_price = pipeline.predict(input_df)
        
        final_price = np.expm1(log_price)[0]
        
        st.divider()
        st.balloons()
        st.subheader("Price Estimation Results")
        
        st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")
        
        st.info(f"""
        **Analysis Summary:**
        - This car has a **Mileage Intensity** of {mileage_intensity} km/year.
        - The model accounts for the popularity of the **{selected_mfr}** brand.
        - Based on current market trends for **{selected_cat}** vehicles.
        """)

    except Exception as e:
        st.error("Prediction Logic Error. This is usually due to a column name mismatch between the app and the model.")
        st.write(e)