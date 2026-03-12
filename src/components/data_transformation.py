import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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
            one_hot_columns = [
                'Category', 'Fuel type',
                'Gear box type', 'Drive wheels',
               'Color'
            ]

            power_columns = [
                'Mileage', 'Age',
                'Engine Volume', 'Levy',
               'Cylinders', 'Mileage_Intensity' 
            ]

            standard_column = ['Airbags']

            binary_columns = ['Leather interior', 'Turbo']

            high_cardinal_columns = ['Manufacturer', 'Model']


            logging.info('Starting Data Transformation according to the requirements of columns')
            standard_transform = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('StandardTransform', StandardScaler())
            ])
        
            one_hot_transform = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('OneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Power Transformer for normalzing the data
            power_transform = Pipeline(steps=[
                ('mputer', SimpleImputer(strategy='median')),
                ('Transformer', PowerTransformer('yeo-johnson'))
            ])

            # Ordinal Transform for converting Yes/No to 1/0
            ordinal_transform = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy='most_frequent')),
                ('Ordinal', OrdinalEncoder(categories=[['No', 'Yes'], ['No', 'Yes']], dtype=int))
            ])
            
            logging.info('Starting Frequency encoding for high cardinals')

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
                        X[col] = X[col].map(self.freq_maps[col]).fillna(0)
                    return X.values


            high_card_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('freq', FrequencyEncoder())
            ])
            

            preprocessor = ColumnTransformer([
                ('StandardT', standard_transform, standard_column),
                ('OneH', one_hot_transform, one_hot_columns),
                ('OrdinalE', ordinal_transform, binary_columns),
                ('PowerT', power_transform, power_columns),
                ('HighC', high_card_pipe, high_cardinal_columns)
            ], remainder='drop') 

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
            target_col = "Price"
            
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