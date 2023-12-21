import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationconfig:
    preprocesser_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    

    def get_data_transformation_object(self):
        try:
            logging.info("Data Tranfermation piple start")
            
            categories_column=['potential_issue','went_on_backorder']
            numreic_column=['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month','sales_1_month', 'min_bank', 'perf_6_month_avg']
           
            num_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='median'))
                ,('scaler', StandardScaler())])

            cate_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='most_frequent')),
                ("Label encoding ",LabelEncoder())
                ,('scaler', StandardScaler())])

            preprocessor = ColumnTransformer([
                ("num_Transfer", num_pipe,numreic_column),
             ("cat_transfer",cate_pipe,categories_column)
             ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:

            logging.info("initiate Data Tranfermation")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            preprocesser_obj=self.get_data_transformation_object()
            
            #Balaceing the data 
            majer_class=train_data[train_data['went_on_backorder']=="No"]
            minar_class=train_data[train_data['went_on_backorder']=="Yes"]
            resample_data=resample(majer_class,n_samples=len(minar_class),random_state=1)
            train_data=pd.concat([resample_data,minar_class])

            columns=['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month','sales_1_month', 'min_bank',
        'potential_issue', 'perf_6_month_avg','went_on_backorder']

            #Feature Extration
            train_data=train_data.loc[:,columns]
            test_data=test_data.loc[:,columns]

            target_col=["went_on_backorder"]
            print(train_data.head())

            #applying the transformation
            train_proceess_data=preprocesser_obj.fit_transform(train_data)
            test_proceess_data=preprocesser_obj.fit_transform(test_data)
            
            #Spilting dependend & independene feature in train data and test data 
            x_train=train_data.drop(target_col,axis=1)
            y_train=train_data[target_col]

            x_test=test_data.drop(target_col,axis=1)
            y_test=test_data[target_col]
            
            print(x_train.head())


            logging.info("Applying preprocessing object on training and testing datasets.")

            print(x_train)
        except Exception as e:
            raise CustomException(e, sys)
    
    
