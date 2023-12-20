from src.exception import CustomException
from src.logger  import logging
from dataclasses import dataclass
import os , sys
import pandas as pd

@dataclass
class DataTransformationconfig:
    preprocesser_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation(self):
    def __init__(self):
        self.data_transformation_config=preprocesser_obj_file_path
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("initiate Data Tranfermation")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            train_data.drop(["sku"],inplace=True,axis=1)
            train_data.drop(["sku"],inplace=True,axis=1)

        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_object(self):
        try:
            logging.info("Data Tranfermation piple start")

        except Exception as e:
            raise CustomException(e, sys)
