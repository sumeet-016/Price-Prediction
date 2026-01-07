import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Custom Frequency Encoder
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X.values

# 2. Data Preparation
data = pd.read_csv('Model_training.csv')

X = data.drop(['Price'], axis=1)
y = np.log1p(data['Price']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Preprocessing Setup
one_hot_cols = ['Category', 'Fuel type', 'Gear box type', 'Drive wheels', 'Color']
binary_cols  = ['Leather interior', 'Turbo']
high_card_cols = ['Manufacturer', 'Model']
num_cols     = ['Mileage', 'Age', 'Engine Volume', 'Levy', 'Cylinders', 'Airbags', 'Mileage_Intensity']

ensemble_transformer = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
    ('onehot', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), one_hot_cols),
    ('binary', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(categories=[['No', 'Yes'], ['No', 'Yes']], dtype=int))]), binary_cols),
    ('high_card', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('freq', FrequencyEncoder())]), high_card_cols)
], remainder='passthrough')

# 4. Initialize Models (No wrappers needed now!)
et_final = ExtraTreesRegressor(n_estimators=300, min_samples_split=5, max_features=1.0, random_state=42)
rf_final = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42)
cat_final = CatBoostRegressor(learning_rate=0.05, l2_leaf_reg=3, iterations=1000, depth=10, silent=True, random_seed=42)

# 5. Final Assembly
final_production_pipe = Pipeline([
    ('preprocess', ensemble_transformer),
    ('model', VotingRegressor(estimators=[('et', et_final), ('rf', rf_final), ('cat', cat_final)]))
])

print("Training final ensemble model on scikit-learn 1.5.2...")
final_production_pipe.fit(X_train, y_train)

with open('final_car_price_model.pkl', 'wb') as f:
    pickle.dump(final_production_pipe, f)

print("SUCCESS: Model saved to final_car_price_model.pkl")