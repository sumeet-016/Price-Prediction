import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



def run_pipeline():
    try:
        logging.info("Pipeline Started")

        # ─── Step 1: Data Ingestion ───────────────────────────────────
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data Transformation Completed | Preprocessor saved at: {preprocessor_path}")


        trainer = ModelTrainer()
        model_path, test_metrics = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(
            f"Model Training Completed | "
            f"R²: {test_metrics['r2']:.4f} | "
            f"MAE: {test_metrics['mae']:,.2f} | "
            f"RMSE: {test_metrics['rmse']:,.2f}"
        )
        logging.info(f"Model saved at: {model_path}")

        logging.info("Pipeline Finished Successfully")

    except Exception as e:
        logging.error("Pipeline Failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_pipeline()