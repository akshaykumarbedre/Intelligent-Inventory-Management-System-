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
            #categories_column=df.dtypes[(df.dtypes!='int64') & (df.dtypes!='float64')].index
           
            num_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='median'))
                ,('scaler', StandardScaler())])

            cate_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='most_frequent')),
                ("Label encoding ",LabelEncoder())
                ,('scaler', StandardScaler())])

            preprocessor = ColumnTransformer([
                ("num_Transfer", num_pipe),
             ("cat_transfer",cate_pipe,)
             ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("initiate Data Tranfermation")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            preprocesser=self.get_data_transformation_object()
            
            #Balaceing the data 
            majer_class=train_data[train_data['went_on_backorder']=="No"]
            minar_class=train_data[train_data['went_on_backorder']=="Yes"]
            resample_data=resample(majer_class,n_samples=len(minar_class),random_state=1)
            train_data=pd.concat([resample_data,minar_class])

            #Droping the column
            #train_data.drop(["sku"],inplace=True,axis=1)
            test_data.drop(["sku"],inplace=True,axis=1)

            target_col=["went_on_backorder"]

            #Spilting dependend & independene feature in train data and test data 
            x_train=train_data.drop(target_col,axis=1)
            y_train=train_data[target_col]

            x_test=test_data.drop(target_col,axis=1)
            y_test=test_data[target_col]
            
            print(train_data.head())
            #applying the transformation
            x_train_proceess_data=preprocesser.fit(x_train)
            x_test_proceess_data=preprocesser.fit_transform(x_test)
            logging.info("Applying preprocessing object on training and testing datasets.")

            print(x_train)

        
        
        except Exception as e:
            raise CustomException(e, sys)