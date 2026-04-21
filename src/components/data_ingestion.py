import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path:  str = os.path.join('artifacts', 'test.csv')
    raw_data_path:   str = os.path.join('artifacts', 'data.csv')

    # ✅ Path relative to project root — works on any machine
    dataset_path:    str = os.path.join('Dataset', 'Dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # ✅ Relative path — portable across machines
            df = pd.read_csv(self.ingestion_config.dataset_path)
            logging.info(f"Raw dataset loaded — shape: {df.shape}")

            initial_count = len(df)

            # ── Price Filter ───────────────────────────────────────────────
            # Remove junk/error entries (<$500) and supercars (>$300k)
            # which skew the general resale pricing model
            df = df[(df['Price'] > 500) & (df['Price'] < 300000)]
            logging.info(f"After price filter   — dropped {initial_count - len(df):,} rows | remaining: {len(df):,}")

            # ── Age Filter ─────────────────────────────────────────────────
            # Exclude vintage vehicles (Age > 30) — different resale dynamics
            count = len(df)
            df = df[df['Age'] <= 30]
            logging.info(f"After age filter     — dropped {count - len(df):,} rows | remaining: {len(df):,}")

            # ── Mileage Filter ─────────────────────────────────────────────
            # Remove unrealistic mileage readings (> 600,000 km)
            count = len(df)
            df = df[df['Mileage'] < 600000]
            logging.info(f"After mileage filter — dropped {count - len(df):,} rows | remaining: {len(df):,}")

            logging.info(f"Final dataset shape after all filters: {df.shape}")

            # ── Save Raw + Split ───────────────────────────────────────────
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,  index=False, header=True)

            logging.info(f"Train size: {len(train_set):,} | Test size: {len(test_set):,}")
            logging.info("Data Ingestion complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)