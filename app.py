import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# =====================================================
# CUSTOM TRANSFORMER (REQUIRED FOR LOADING MODEL)
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
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps.get(col, {})).fillna(0)
        return X.values

# =====================================================
# LOAD MODEL + DATA
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_car_price_model.pkl")
    df = pd.read_csv("car_price_prediction_updated.csv")

    # clean strings
    for col in [
        'Manufacturer','Model','Category','Leather interior','Fuel type',
        'Gear box type','Drive wheels','Wheel','Color'
    ]:
        df[col] = df[col].astype(str).str.strip()

    df['Manufacturer'] = df['Manufacturer'].str.title()
    return model, df

pipeline, data = load_assets()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.title("🚗 Car Price Predictor")
    st.markdown("Predict **used car market price** using ML.")
    st.markdown("---")
    st.markdown("**Model:** Random Forest + Pipeline")
    st.markdown("**Encoding:** Frequency Encoding")
    st.markdown("---")
    st.markdown("👨‍💻 **Sumeet Kumar Pal**")
    st.markdown("[GitHub](https://github.com/sumeet-016)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/palsumeet/)")

# =====================================================
# MAIN UI
# =====================================================
st.title("🚘 Used Car Price Prediction")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ---------------- MANUFACTURER ----------------
    with col1:
        manufacturers = sorted(data['Manufacturer'].unique())
        selected_mfr = st.selectbox(
            "Manufacturer",
            manufacturers,
            key="manufacturer"
        )

    # ---------------- MODEL (FIXED) ----------------
    with col2:
        filtered_models = sorted(
            data[data['Manufacturer'] == selected_mfr]['Model'].unique()
        )

        # 🔥 FORCE RESET WHEN MANUFACTURER CHANGES
        if "prev_mfr" not in st.session_state:
            st.session_state.prev_mfr = selected_mfr

        if st.session_state.prev_mfr != selected_mfr:
            st.session_state.model = filtered_models[0]
            st.session_state.prev_mfr = selected_mfr

        selected_model = st.selectbox(
            "Model",
            filtered_models,
            key="model"
        )

    # ---------------- CATEGORY ----------------
    with col3:
        category = st.selectbox(
            "Category",
            sorted(data['Category'].unique())
        )

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:
        fuel = st.selectbox("Fuel type", sorted(data['Fuel type'].unique()))
        gear = st.selectbox("Gear box type", sorted(data['Gear box type'].unique()))

    with col5:
        drive = st.selectbox("Drive wheels", sorted(data['Drive wheels'].unique()))
        wheel = st.selectbox("Wheel", sorted(data['Wheel'].unique()))

    with col6:
        leather = st.selectbox("Leather interior", ["Yes", "No"])
        color = st.selectbox("Color", sorted(data['Color'].unique()))

    st.divider()

    col7, col8, col9 = st.columns(3)

    with col7:
        levy = st.number_input("Levy", value=float(data['Levy'].median()))
        mileage = st.number_input("Mileage", value=int(data['Mileage'].median()))

    with col8:
        cylinders = st.number_input("Cylinders", value=float(data['Cylinders'].median()))
        airbags = st.slider("Airbags", 0, 16, int(data['Airbags'].median()))

    with col9:
        age = st.number_input("Age (years)", 0, 100, int(data['Age'].median()))

    submit = st.form_submit_button("🔮 Predict Price")

# =====================================================
# PREDICTION
# =====================================================
if submit:
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
        log_price = pipeline.predict(input_df)
        final_price = np.expm1(log_price)[0]
        st.success(f"### 💰 Estimated Price: ${final_price:,.2f}")
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
