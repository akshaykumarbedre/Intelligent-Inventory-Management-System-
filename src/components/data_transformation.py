import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,target_column_Encoding,outlier_remover

from sklearn.compose import ColumnTransformer
import numpy as np

@dataclass
class DataTransformationconfig:
    preprocesser_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Tranfermation piple start")
            
            num_pipe = Pipeline(
                [("handle missing value", SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
                 ])

            return num_pipe
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:

            logging.info("initiate Data Tranfermation")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            train_data=outlier_remover(train_data)
            test_data=outlier_remover(test_data)

            # Balaceing train_data the data
            majer_class = train_data[train_data['went_on_backorder'] == "No"]
            minar_class = train_data[train_data['went_on_backorder'] == "Yes"]
            resample_data = resample(
                majer_class, n_samples=len(minar_class), random_state=1)
            train_data = pd.concat([resample_data, minar_class])

            # Balaceing train_data the data
            majer_class = test_data[test_data['went_on_backorder'] == "No"]
            minar_class = test_data[test_data['went_on_backorder'] == "Yes"]
            resample_data = resample(
                majer_class, n_samples=len(minar_class), random_state=1)
            test_data = pd.concat([resample_data, minar_class])

            columns = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'sales_1_month', 'min_bank', 'perf_6_month_avg', 'went_on_backorder']

            # Feature Extration
            train_data = train_data.loc[:, columns]
            test_data = test_data.loc[:, columns]

            target_col = ["went_on_backorder"]


            # Spilting dependend & independene feature in train data and test data
            x_train = train_data.drop(target_col, axis=1)
            y_train = train_data[target_col]
            
            x_test = test_data.drop(target_col, axis=1)
            y_test = test_data[target_col]


            #Encoding target column 
            y_train=target_column_Encoding(y_train)
            y_test=target_column_Encoding(y_test)

            preprocesser_obj = self.get_data_transformation_object()
        
            x_train_Procese = preprocesser_obj.fit_transform(x_train)
            x_test_Procese = preprocesser_obj.fit_transform(x_test)

            train_arr = np.c_[x_train_Procese, np.array(y_train)]
            test_arr = np.c_[x_test_Procese, np.array(y_test)]

            logging.info("Applying preprocessing object on training and testing datasets.")
            save_object(self.data_transformation_config.preprocesser_obj_file_path,preprocesser_obj)

            return train_arr,test_arr,self.data_transformation_config.preprocesser_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
            

    
