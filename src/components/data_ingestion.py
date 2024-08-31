import os
import sys
from dataclasses import dataclass

import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:

    data_path: str = os.path.join("artifacts", "Data.csv")
    train_data_path: str = os.path.join("artifacts", "Train_data.csv")
    test_data_path: str = os.path.join("artifacts", "Test_data.csv")


class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method start")

        try:
            Raw_data = pd.read_csv(self.ingestion_config.data_path)
            train_data, test_data =train_test_split(Raw_data)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            train_data = pd.read_csv(self.ingestion_config.train_data_path)
            test_data = pd.read_csv(self.ingestion_config.test_data_path)



            logging.info("Data Ingestion competed")
            logging.info(
                f"Data Ingestion competed train data \n{str(train_data.head())}")
            logging.info(
                f"Data Ingestion competed test data \n{str(test_data.head())}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred {e} {sys}")
            raise ConnectionResetError(sys, e)

