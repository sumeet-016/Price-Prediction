import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from src.exception import CustomException
from src.logger import logging
from src.components.feature_engineering import FeatureEngine
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Returns only the encoding/imputing preprocessor — NO FeatureEngine here."""
        try:
            numeric_features = [
                'Levy', 'Mileage', 'Age',
                'Engine Volume', 'Mileage_Intensity',
                'Airbags', 'is_levy_zero'
            ]
            one_hot_features = [
                'Category', 'Fuel type',
                'Gear box type', 'Color',
                'Inventory_Segment'
            ]
            binary_features = [
                'Leather interior', 'Turbo',
                'Is_Premium_Brand'
            ]
            high_cardinal = ['Manufacturer']

            logging.info('Building preprocessing pipeline')

            nums_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])

            one_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ))
            ])

            binary_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(
                    categories=[
                        ['No', 'Yes'],
                        ['False', 'True'],
                        ['False', 'True']
                    ]
                ))
            ])

            high_card_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('Target_Enco', TargetEncoder(
                    cv=3,
                    target_type='continuous'
                ))
            ])

            # ✅ Just the ColumnTransformer — no FeatureEngine wrapper
            preprocessor = ColumnTransformer([
                ('num',       nums_pipe,     numeric_features),
                ('onehot',    one_pipe,      one_hot_features),
                ('binary',    binary_pipe,   binary_features),
                ('high_card', high_card_pipe, high_cardinal)
            ])

            logging.info('Preprocessing pipeline built successfully')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            target_col = 'Price'

            # ─── Step 1: Feature Engineering on full df ───────────────────
            # Applied on full df so row drops keep X and y in sync
            feature_engine = FeatureEngine()
            train_df = feature_engine.fit_transform(train_df)
            test_df  = feature_engine.transform(test_df)

            logging.info(f"After FeatureEngine — Train: {train_df.shape}, Test: {test_df.shape}")

            # ─── Step 2: Split X and y ────────────────────────────────────
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].astype(float)

            X_test  = test_df.drop(columns=[target_col])
            y_test  = test_df[target_col].astype(float)

            # ─── Step 3: Fit the preprocessor ────────────────────────────
            preprocessing_obj = self.get_data_transformer_object()
            train_arr_processed = preprocessing_obj.fit_transform(X_train, y_train)
            test_arr_processed  = preprocessing_obj.transform(X_test)

            # ─── Step 4: Combine features + target ───────────────────────
            train_arr = np.c_[train_arr_processed, np.array(y_train)]
            test_arr  = np.c_[test_arr_processed,  np.array(y_test)]

            # ─── Step 5: Bundle into one pipeline and save ───────────────
            # feature_engine and preprocessing_obj are ALREADY fitted above
            # Wrapping them means inference needs only ONE pkl file
            full_pipeline = Pipeline(steps=[
                ('feature_engine', feature_engine),
                ('preprocessing',  preprocessing_obj)
            ])

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=full_pipeline
            )

            logging.info("Saved full pipeline (feature_engine + preprocessor) to preprocessor.pkl")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)