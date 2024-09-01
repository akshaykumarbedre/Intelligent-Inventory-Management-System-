# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'RandomForestClassifier':RandomForestClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'SGDClassifier':SGDClassifier(),
            'KNeighborsClassifier':KNeighborsClassifier(),
            "GradientBoostingClassifier":GradientBoostingClassifier()
                                                                     
        }
            
            model_report:dict=self.evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model = models[best_model_name]
            result=f'Best Model Found , Model Name : {best_model_name} , Accuary is : {round(best_model_score*100,2)}'
            print(result)
            print('\n====================================================================================\n')
            logging.info(f'{result}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            return result
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

    def evaluate_model(self,X_train,y_train,X_test,y_test,models):
        try:
            report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                # Train model
                model.fit(X_train,y_train)

                # Predict Testing data
                y_test_pred =model.predict(X_test)

                # Get R2 scores for train and test data

                test_model_score = f1_score(y_test,y_test_pred)

                report[list(models.keys())[i]] =  test_model_score

            return report
        
        except Exception as e:
                logging.info('Exception occured during model training')
                raise CustomException(e,sys)
