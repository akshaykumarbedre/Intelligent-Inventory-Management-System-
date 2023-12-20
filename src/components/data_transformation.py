from src.exception import CustomException
from src.logger  import logging
from dataclasses import dataclass
import os , sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

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

            preprocesser=get_data_transformation_object()
            
            #Balaceing the data 
            majer_class=train_data[train_data['went_on_backorder']=="No"]
            minar_class=train_data[train_data['went_on_backorder']=="Yes"]
            resample_data=resample(majer_class,n_samples=len(minar_class),random_state=1)
            train_data=pd.concat([resample_data,minar_class])

            #Droping the column
            train_data.drop(["sku"],inplace=True,axis=1)
            test_data.drop(["sku"],inplace=True,axis=1)

            preprocesser()

        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_object(self):
        try:
            logging.info("Data Tranfermation piple start")
            #categories_column=df.dtypes[(df.dtypes!='int64') & (df.dtypes!='float64')].index
           
            num_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='median'))
                ,('scaler', StandardScaler())])

            cate_pipe = Pipeline(
                [("handle missing value",SimpleImputer(strategy='most_frequent'))
                ("Label encoding ",LabelEncoder())
                ,('scaler', StandardScaler())])

            preprocessor = ColumnTransformer([
                ("num_Transfer", num_pipe),
             ("cat_transfer",cate_pipe,)
             ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
