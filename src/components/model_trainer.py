import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            y_train = y_train.astype(int)

            et = ExtraTreesRegressor(
                max_depth=30, max_features=1,
                min_samples_leaf=1, min_samples_split=3,
                n_estimators=229, random_state=42
            )

            lgb = LGBMRegressor(
                learning_rate=0.1, max_depth=30,
                n_estimators=363, num_leaves=66,
                random_state=42
            )

            cb = CatBoostRegressor(
                depth=8, iterations=1114,
                l2_leaf_reg=3, learning_rate=0.5,
                random_state=42
            )

            ensemble = StackingRegressor(
                estimators=[
                    ('et', et),
                    ('lgb', lgb),
                    ('cb', cb)
                ],
                final_estimator=Ridge(),
                cv=3,
                n_jobs=-1
            )

            logging.info("Training ensemble model")
            ensemble.fit(X_train, y_train)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=ensemble)
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)