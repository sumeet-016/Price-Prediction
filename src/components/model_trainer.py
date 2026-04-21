import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

    def evaluate_model(self, y_true, y_pred, split_name="Test"):
        """Log and return key regression metrics."""
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        logging.info(f"── {split_name} Metrics ──────────────────")
        logging.info(f"  R²   : {r2:.4f}")
        logging.info(f"  MAE  : {mae:,.2f}")
        logging.info(f"  RMSE : {rmse:,.2f}")

        return {"r2": r2, "mae": mae, "rmse": rmse}

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )


            y_train = y_train.astype(float)
            y_test  = y_test.astype(float)

            et = ExtraTreesRegressor(
                max_depth=30,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=3,
                n_estimators=229,
                random_state=42,
                n_jobs=-1
            )

            lgb = LGBMRegressor(
                learning_rate=0.1,
                max_depth=30,
                n_estimators=363,
                num_leaves=66,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            cb = CatBoostRegressor(
                depth=8,
                iterations=1114,
                l2_leaf_reg=3,
                learning_rate=0.05,
                random_state=42,
                verbose=0
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

            logging.info("Training ensemble model...")
            ensemble.fit(X_train, y_train)
            logging.info("Training complete")

            # ✅ Evaluate on both splits to detect overfitting
            train_preds = ensemble.predict(X_train)
            test_preds  = ensemble.predict(X_test)

            train_metrics = self.evaluate_model(y_train, train_preds, split_name="Train")
            test_metrics  = self.evaluate_model(y_test,  test_preds,  split_name="Test")

            # ✅ Warn if model is overfitting
            if train_metrics["r2"] - test_metrics["r2"] > 0.1:
                logging.warning(
                    f"Possible overfitting detected — "
                    f"Train R²: {train_metrics['r2']:.4f}, "
                    f"Test R²: {test_metrics['r2']:.4f}"
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble
            )

            return self.model_trainer_config.trained_model_file_path, test_metrics

        except Exception as e:
            raise CustomException(e, sys)