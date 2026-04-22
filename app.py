import sys
import os

# ✅ Add project root to path so Python can find predict_pipeline.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor")

# ─── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        manufacturer = st.selectbox("Manufacturer", [
            "TOYOTA", "HYUNDAI", "FORD", "BMW",
            "MERCEDES-BENZ", "KIA", "HONDA", "CHEVROLET"
        ])
        category = st.selectbox("Category", [
            "Sedan", "Jeep", "Hatchback",
            "Minivan", "Coupe", "Universal"
        ])
        fuel_type = st.selectbox("Fuel Type", [
            "Petrol", "Diesel", "Hybrid",
            "CNG", "Plug-in Hybrid"
        ])
        gear_box_type = st.selectbox("Gear Box", [
            "Automatic", "Manual",
            "Tiptronic", "Variator"
        ])
        color = st.selectbox("Color", [
            "Black", "White", "Silver",
            "Grey", "Blue", "Red", "Green"
        ])

    with col2:
        mileage       = st.number_input("Mileage (km)",        min_value=0,   max_value=600000, value=50000)
        engine_volume = st.number_input("Engine Volume (L)",   min_value=0.0, max_value=6.0,    value=2.0, step=0.1)
        age           = st.number_input("Car Age (years)",     min_value=0,   max_value=30,     value=5)
        airbags       = st.number_input("Airbags",             min_value=0,   max_value=16,     value=4)
        levy          = st.number_input("Levy",                min_value=0,   max_value=5000,   value=0)

    with col3:
        leather_interior  = st.selectbox("Leather Interior",   ["Yes", "No"])
        turbo             = st.selectbox("Turbo",              ["True", "False"])
        is_premium_brand  = st.selectbox("Premium Brand",      [False, True])
        inventory_segment = st.selectbox("Inventory Segment",  [
            "Budget", "Mid-Range", "Premium", "Luxury"
        ])

    submitted = st.form_submit_button("Predict Price 💰")

# ─── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    try:
        # Step 1 — collect inputs into CustomData
        data = CustomData(
            manufacturer      = manufacturer,
            category          = category,
            fuel_type         = fuel_type,
            gear_box_type     = gear_box_type,
            color             = color,
            leather_interior  = leather_interior,
            turbo             = turbo,
            mileage           = mileage,
            engine_volume     = engine_volume,
            age               = age,
            airbags           = airbags,
            levy              = levy,
            is_premium_brand  = is_premium_brand,
            inventory_segment = inventory_segment,
        )

        # Step 2 — convert to DataFrame
        input_df = data.get_data_as_dataframe()

        # Step 3 — predict
        pipeline = PredictPipeline()
        price    = pipeline.predict(input_df)

        st.success(f"### Estimated Car Price: **${price:,.0f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")