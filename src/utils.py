import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

def outlier_remover(df):
    try:
        columns = df.select_dtypes(include=['int64', 'float64'])
        n_std = 3
        # For each column, remove rows that are more than n_std standard deviations away from the mean
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)]

        return df

    except Exception as e:
            logging.info('Exception Occured in outlier remover function utils')
            raise CustomException(e,sys)

def target_column_Encoding(df):
    try:
        df=df["went_on_backorder"]
        df=df.str.replace("Yes","1")
        df=df.str.replace("No","0")
        df=df.astype(int)

        return df
        
    except Exception as e:
        logging.info('Exception Occured in target_column_Encoding utils')
        raise CustomException(e, sys)