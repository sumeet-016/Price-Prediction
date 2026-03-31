import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.feature_engineering import FeatureEngine
from components.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException


def run_pipeline():
    try:
        logging.info("🚀 Pipeline Started")

        # Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        # Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data Transformation Completed")

        # Feature Engineering
        feature_eng = FeatureEngine()
        train_arr = feature_eng.transform(train_arr)
        test_arr = feature_eng.transform(test_arr)
        logging.info("Feature Engineering Completed")

        # Model Training
        trainer = ModelTrainer()
        model_score = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training Completed | Score: {model_score}")

        logging.info("Pipeline Finished Successfully")

    except Exception as e:
        logging.error("Pipeline Failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()