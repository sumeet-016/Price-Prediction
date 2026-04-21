import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self):
        # All learned stats stored here during fit()
        self.mfr_to_keep_ = None
        self.median_mileage_per_age_ = None
        self.median_levy_per_age_ = None
        self.median_airbags_ = None
        self.median_engine_vol_ = None
        self.global_mileage_rate_ = None

    def fit(self, X, y=None):
        """Learn all statistics from TRAINING data only."""
        try:
            data = X.copy()

            # ✅ Learn which manufacturers to keep (from train only)
            mfr_counts = data['Manufacturer'].value_counts()
            self.mfr_to_keep_ = mfr_counts[mfr_counts >= 5].index

            # ✅ Learn median Levy per Age group (from train only)
            self.median_levy_per_age_ = (
                data[data['Levy'] > 0]
                .groupby('Age')['Levy']
                .median()
            )

            # ✅ Learn global mileage rate (from train only)
            self.global_mileage_rate_ = (
                data[data['Mileage'] > 0]['Mileage'] / data['Age']
            ).median()

            # ✅ Learn median airbags per Manufacturer+Category (from train only)
            self.median_airbags_ = (
                data[data['Airbags'] > 0]
                .groupby(['Manufacturer', 'Category'])['Airbags']
                .median()
            )

            # ✅ Learn median Engine Volume per Manufacturer (from train only)
            self.median_engine_vol_ = (
                data[data['Engine_Volume_Num'] > 0]
                .groupby('Manufacturer')['Engine_Volume_Num']
                .median()
            )

            logging.info("FeatureEngine fit complete — all stats learned from training data")
            return self

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X, y=None):
        """Apply learned stats — no recomputation from current data."""
        try:
            logging.info("Feature Engineering started")
            data = X.copy()

            # ── Manufacturer Cleaning ──────────────────────────────────────
            data['Manufacturer'] = data['Manufacturer'].str.upper().str.strip()

            # ✅ Use manufacturers learned from train — don't recompute
            data = data[data['Manufacturer'].isin(self.mfr_to_keep_)]
            data.reset_index(drop=True, inplace=True)

            # ── Outlier Removal ────────────────────────────────────────────
            data = data[data['Cylinders'] <= 12]

            # ── Column Renaming & Dropping ─────────────────────────────────
            data = data.drop(columns=['Engine volume'], axis=1)
            data.rename(columns={'Engine_Volume_Num': 'Engine Volume'}, inplace=True)

            # ── Feature Creation ───────────────────────────────────────────
            data['Engine_Efficiency'] = data['Engine Volume'] / data['Cylinders']
            data['is_levy_zero'] = (data['Levy'] == 0).astype(int)

            # ── Fix Levy using TRAIN medians ───────────────────────────────
            data['Levy'] = data['Levy'].astype(float)
            data['Levy'] = data['Levy'].replace(0, np.nan)
            data['Levy'] = data['Age'].map(self.median_levy_per_age_)

            # ── Fix Mileage using TRAIN rate ───────────────────────────────
            data['Mileage'] = data['Mileage'].astype(float)
            data.loc[data['Mileage'] == 0, 'Mileage'] = (
                data['Age'] * self.global_mileage_rate_
            )
            # ── Fix Airbags using TRAIN medians ───────────────────────────
            def fix_airbags(row):
                if row['Airbags'] == 0:
                    key = (row['Manufacturer'], row['Category'])
                    return self.median_airbags_.get(key, 4)
                return row['Airbags']

            data['Airbags'] = data.apply(fix_airbags, axis=1)

            # ── Fix Engine Volume using TRAIN medians ──────────────────────
            def fix_engine_vol(row):
                if row['Engine Volume'] == 0:
                    return self.median_engine_vol_.get(row['Manufacturer'], 2.0)
                return row['Engine Volume']

            data['Engine Volume'] = data.apply(fix_engine_vol, axis=1)

            # ── Final Features ─────────────────────────────────────────────
            data['Mileage_Intensity'] = (data['Mileage'] / (data['Age'] + 1)).round(2)
            data[['Is_Premium_Brand', 'Turbo']] = data[['Is_Premium_Brand', 'Turbo']].astype(str)

            data = data.drop(columns=[
                'Prod. year', 'Cylinders',
                'Price_Per_Litre', 'Wheel',
                'Doors', 'Age_Group',
                'Mileage_Band', 'Model'
            ])

            logging.info("Feature Engineering Completed")
            return data

        except Exception as e:
            raise CustomException(e, sys)