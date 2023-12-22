import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            #feature #['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'sales_1_month', 'min_bank', 'perf_6_month_avg', 'went_on_backorder']
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,national_inv:float,lead_time:float,in_transit_qty:float,forecast_3_month:float,sales_1_month:float,min_bank:float,perf_6_month_avg:float):        
        self.national_inv=national_inv
        self.lead_time=lead_time
        self.in_transit_qty=in_transit_qty
        self.forecast_3_month=forecast_3_month
        self.sales_1_month=sales_1_month
        self.min_bank=min_bank
        self.perf_6_month_avg = perf_6_month_avg


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "national_inv":[self.national_inv],
            "lead_time":[self.lead_time],
            "in_transit_qty":[self.in_transit_qty],
            "forecast_3_month":[self.forecast_3_month],
            "sales_1_month":[self.sales_1_month],
            "min_bank":[self.min_bank],
            "perf_6_month_avg":[self.perf_6_month_avg]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

#cd=CustomData(national_inv=24,lead_time=8,in_transit_qty=0,forecast_3_month=3456,sales_1_month=10,min_bank=7,perf_6_month_avg=1)
#df=cd.get_data_as_dataframe()
#
#pp=PredictPipeline()
#print(pp.predict(df))
