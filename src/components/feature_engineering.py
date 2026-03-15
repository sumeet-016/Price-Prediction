import sys
import pandas as pd
import re
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            logging.info("Feature Engineering started")
            data = X.copy()
            
            # Removing the rows which have Car Age higher than 30
            data = data[data['Age'] <= 30]

            # Remove car whose km is higher than 600,000
            data = data[data['Mileage'] < 600000]

            data = data[data['Levy'] < 4500]
            data = data['Levy'].round(2)

            # Removing manufactures less than 10
            mfr_count = data['Manufacture'].value_count()
            mfr_to_keep = mfr_count[mfr_count > 10].index
            data = data[data['Manufacture'].isin(mfr_to_keep)]

            data.reset_index(drop=True, inplace=True)

            # Normalize car model names to ensure consistent categorical encoding,
            # then filter out low-frequency models to minimize sparsity,
            # reduce overfitting, and improve downstream model generalization
            def clean_model_name(text):
                if pd.isna(text):
                    return text
                text = str(text).upper().strip()
                text = re.sub(r'[^A-Z0-9 ]', ' ', text)
                text = " ".join(text.split())
                return text

            data['Model'] = data['Model'].apply(clean_model_name)

            data['Model'] = data['Model'].astype(str).str.strip().str.upper()

            model_counts = data['Model'].value_counts()
            models_to_keep = model_counts[model_counts > 2].index

            data = data[data['Model'].isin(models_to_keep)].reset_index(drop=True)

            data['Manufacturer'] = data['Manufacturer'].str.upper().str.strip()

            # Feature engineering for engine specifications:
            # separates engine capacity as a continuous variable and turbo presence
            # as a binary feature to capture both engine size and performance influence
            def clean_engine_volume(value):
                value = str(value).lower()
                is_turbo = 'Yes' if 'turbo' in value else 'No'
                
                numeric_part = re.findall(r"[-+]?\d*\.\d+|\d+", value)
                volume = float(numeric_part[0]) if numeric_part else 0.0
                return volume, is_turbo

            data[['Engine_Volume', 'Is_Turbo']] = data['Engine volume'].apply(
                lambda x: pd.Series(clean_engine_volume(x))
            )

            # Remove unrealistic or extreme cylinder values to eliminate outliers
            data = data[data['Cylinders'] <= 12]

            data = data.drop(columns=['Engine volume', 'Wheel'], axis=1)

            data.rename(columns={
                'Engine_Volume': 'Engine Volume',
                'Is_Turbo': 'Turbo'
            }, inplace=True)

            # Fix Mileage zeros using car Age
            median_mileage = (data[data['Mileage'] > 0]['Mileage'] / data['Age']).median()
            data.loc[data['Mileage'] == 0, 'Mileage'] = data['Age'] * median_mileage

            # Fix Airbags zeros using Manufacturer/Category standards
            data['Airbags'] = data.groupby(['Manufacturer', 'Category'])['Airbags'].transform(
                lambda x: x.replace(0, x[x > 0].median() if not x[x > 0].empty else 4)
            )

            # Replace 0 in Engine Volume using the median of the same Manufacturer and Model
            data['Engine Volume'] = data.groupby(['Manufacturer', 'Model'])['Engine Volume'].transform(
                lambda x: x.replace(0, x[x > 0].median() if not x[x > 0].empty else 2.0)
            )
            data['Mileage_Intensity'] = (data['Mileage'] / (data['Age'] + 1)).round(2)


            logging.nfo("Feature Engineering Completed")

            return data
        except Exception as e:
            raise CustomException(e, sys)