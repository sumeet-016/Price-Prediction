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

            # Remove manufacturers > 5
            mfr_counts = data['Manufacturer'].value_counts()
            mfr_to_keep = mfr_counts[mfr_counts >= 5].index
            data = data[data['Manufacturer'].isin(mfr_to_keep)]

            data.reset_index(drop=True, inplace=True)


            data['Manufacturer'] = data['Manufacturer'].str.upper().str.strip()


            # Remove unrealistic or extreme cylinder values to eliminate outliers
            data = data[data['Cylinders'] <= 12]

            data = data.drop(columns=['Engine volume'], axis=1)

            data.rename(columns={
                'Engine_Volume_Num': 'Engine Volume'
            }, inplace=True)


            data['Engine_Efficiency'] = (data['Engine Volume'])/(data['Cylinders'])

            data['is_levy_zero'] = (data['Levy'] == 0).astype(int)


            # Step 1: Treat 0 as missing
            data['Levy'] = data['Levy'].replace(0, np.nan)

            # Step 2: Fill using median of same Age group
            data['Levy'] = data.groupby('Age')['Levy'].transform(lambda x: x.fillna(x.median()))


            # Fix Mileage zeros using car Age
            median_mileage = (data[data['Mileage'] > 0]['Mileage'] / data['Age']).median()
            data.loc[data['Mileage'] == 0, 'Mileage'] = data['Age'] * median_mileage

            # Fix Airbags zeros using Manufacturer/Category standards
            data['Airbags'] = data.groupby(['Manufacturer', 'Category'])['Airbags'].transform(
                lambda x: x.replace(0, x[x > 0].median() if not x[x > 0].empty else 4)
            )

            # Replace 0 in Engine Volume using the median of the same Manufacturer and Model
            data['Engine Volume'] = data.groupby(['Manufacturer'])['Engine Volume'].transform(
                lambda x: x.replace(0, x[x > 0].median() if not x[x > 0].empty else 2.0)
            )
            data['Mileage_Intensity'] = (data['Mileage'] / (data['Age'] + 1)).round(2)

            data[['Is_Premium_Brand', 'Turbo']] = data[['Is_Premium_Brand', 'Turbo']].astype(str)

            data = data.drop(columns=[
                'Prod. year', 'Cylinders',
                'Price_Per_Litre', 'Wheel',
                'Doors','Age_Group',
                'Mileage_Band','Model'
            ])

            logging.info("Feature Engineering Completed")

            return data
        except Exception as e:
            raise CustomException(e, sys)