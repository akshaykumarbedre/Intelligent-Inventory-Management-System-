from src.exception import CustomException
from src.logger  import logging
from dataclasses import dataclass
import os , sys
import pandas as pd

@dataclass
class DataingestionConfig:
    train_data_path=os.path.join("artifacts","train_data.csv")
    test_data_path=os.path.join("artifacts","test_data.csv")

class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataingestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method start")

        try:
            train_data=pd.read_csv(self.ingestion_config.train_data_path)
            test_data=pd.read_csv(self.ingestion_config.test_data_path)

            

            logging.info("Data Ingestion competed")
            logging.info(f"Data Ingestion competed train data \n{str(train_data.head())}")
            logging.info(f"Data Ingestion competed test data \n{str(test_data.head())}")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise ConnectionResetError(sys,e)
            logging.error(f"Errror occured occured {e}   {sys}")





