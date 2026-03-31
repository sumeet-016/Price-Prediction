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
        try:

            # Tranformation of each column accordling for model training
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

            binary_features =[
                'Leather interior', 'Turbo',
                'Is_Premium_Brand'
            ]

            high_cardinal = ['Manufacturer']


            logging.info('Starting Data Transformation according to the requirements of columns')
            
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
                ('Target_Enc', TargetEncoder())
            ])
            

            preprocessor = ColumnTransformer([
                # No Scaling needed for Trees
                ('num', nums_pipe, numeric_features),

                # Categorical encoding
                ('onehot', one_pipe, one_hot_features),

                # Binary encoding (Leather and Turbo)
                ('binary', binary_pipe, binary_features),

                # Frequency encoding for high cardinality
                ('high_card', high_card_pipe, high_cardinal)
                ])

            logging.info('Data Transformation completed')

            return Pipeline(steps=[
                ('feature_engine', FeatureEngine()),
                ('preprocessing', preprocessor)
            ])

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()
            target_col = 'Price'
            
            X_train = train_df.drop(columns=[target_col], axis=1)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col], axis=1)
            y_test = test_df[target_col]

            train_arr_processed = preprocessing_obj.fit_transform(X_train)
            test_arr_processed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[train_arr_processed, np.array(y_train)]
            test_arr = np.c_[test_arr_processed, np.array(y_test)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)